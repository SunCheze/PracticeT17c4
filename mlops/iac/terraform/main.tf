###############################################################################
# Terraform-модуль для облачного варианта ML-стека Entity Resolution.
#
# Заменяет docker-compose.yml для случая, когда нужно вынести инфраструктуру
# из on-prem в облако (AWS как пример). Та же логика, другие провайдеры:
#   - PostgreSQL  -> RDS
#   - MinIO       -> S3
#   - serving     -> ECS Fargate за ALB
#   - Airflow     -> MWAA или контейнерный ECS
#   - Prometheus  -> Managed Prometheus или контейнер
#
# Модули в этом файле сознательно сжаты до минимума, чтобы показать архитектуру
# без шума IAM-политик и security groups. Полная версия — в подкаталогах.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.70" }
  }
  backend "s3" {
    bucket = "er-mlops-tfstate"
    key    = "prod/terraform.tfstate"
    region = "eu-central-1"
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { Project = "entity-resolution", ManagedBy = "terraform" } }
}

###############################################################################
# VPC + сетка
###############################################################################
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  name    = "er-mlops-vpc"
  cidr    = "10.0.0.0/16"

  azs              = ["${var.region}a", "${var.region}b"]
  private_subnets  = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets   = ["10.0.101.0/24", "10.0.102.0/24"]
  enable_nat_gateway = true
}

###############################################################################
# Хранилища: RDS (Postgres) + S3 (артефакты модели + DVC remote)
###############################################################################
resource "aws_db_instance" "postgres" {
  identifier             = "er-mlops-pg"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.t4g.medium"
  allocated_storage      = 50
  storage_encrypted      = true
  username               = "mlops"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  backup_retention_period = 7
  skip_final_snapshot    = false
  deletion_protection    = true
}

resource "aws_db_subnet_group" "this" {
  name       = "er-mlops-pg-subnets"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "er-mlops-artifacts-${var.environment}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

###############################################################################
# Контейнерные сервисы: ECS Fargate
###############################################################################
resource "aws_ecs_cluster" "this" {
  name = "er-mlops"
  setting { name = "containerInsights"; value = "enabled" }
}

# Serving production
module "serving_production" {
  source        = "./modules/ecs_service"
  cluster_arn   = aws_ecs_cluster.this.arn
  service_name  = "serving-production"
  image         = "${var.ecr_uri}/er-serving:${var.production_tag}"
  port          = 8000
  cpu           = 1024
  memory        = 2048
  desired_count = 2
  environment = {
    MODEL_STAGE         = "Production"
    MLFLOW_TRACKING_URI = "https://${aws_lb.mlflow.dns_name}"
  }
  alb_target_group_arn = aws_lb_target_group.production.arn
  subnets              = module.vpc.private_subnets
  security_groups      = [aws_security_group.serving.id]
}

# Serving canary — тот же модуль, другая ECR-tag и target group
module "serving_canary" {
  source        = "./modules/ecs_service"
  cluster_arn   = aws_ecs_cluster.this.arn
  service_name  = "serving-canary"
  image         = "${var.ecr_uri}/er-serving:${var.canary_tag}"
  port          = 8000
  cpu           = 1024
  memory        = 2048
  desired_count = 1
  environment = {
    MODEL_STAGE         = "Staging"
    MLFLOW_TRACKING_URI = "https://${aws_lb.mlflow.dns_name}"
  }
  alb_target_group_arn = aws_lb_target_group.canary.arn
  subnets              = module.vpc.private_subnets
  security_groups      = [aws_security_group.serving.id]
}

###############################################################################
# ALB c weighted target groups — заменяет nginx из on-prem варианта
###############################################################################
resource "aws_lb" "serving" {
  name               = "er-serving-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = module.vpc.public_subnets
  security_groups    = [aws_security_group.alb.id]
}

resource "aws_lb_target_group" "production" {
  name        = "er-serving-production"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"
  health_check { path = "/health"; interval = 15; timeout = 3 }
}

resource "aws_lb_target_group" "canary" {
  name        = "er-serving-canary"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = module.vpc.vpc_id
  target_type = "ip"
  health_check { path = "/health"; interval = 15; timeout = 3 }
}

# Weighted forwarding — главный артефакт для blue/green-промоушена.
# Значения weight меняются через terraform apply с другим .tfvars
# или через AWS CLI из Airflow-DAG'а promote_model.
resource "aws_lb_listener" "serving" {
  load_balancer_arn = aws_lb.serving.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.acm_certificate_arn

  default_action {
    type = "forward"
    forward {
      target_group { arn = aws_lb_target_group.production.arn; weight = var.weight_production }
      target_group { arn = aws_lb_target_group.canary.arn;     weight = var.weight_canary }
      stickiness   { enabled = false; duration = 60 }
    }
  }
}

###############################################################################
# Outputs
###############################################################################
output "serving_url"    { value = "https://${aws_lb.serving.dns_name}" }
output "rds_endpoint"   { value = aws_db_instance.postgres.endpoint; sensitive = true }
output "artifacts_bucket" { value = aws_s3_bucket.artifacts.id }
