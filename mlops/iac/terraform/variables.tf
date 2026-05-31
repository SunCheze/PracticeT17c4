variable "region" {
  description = "AWS-регион"
  type        = string
  default     = "eu-central-1"
}

variable "environment" {
  description = "production | staging"
  type        = string
}

variable "db_password" {
  description = "Пароль RDS (передаётся через TF_VAR или Secrets Manager)"
  type        = string
  sensitive   = true
}

variable "ecr_uri" {
  description = "ECR-репозиторий с образами сервинга"
  type        = string
}

variable "production_tag" {
  description = "Docker tag для production-реплики"
  type        = string
  default     = "stable"
}

variable "canary_tag" {
  description = "Docker tag для canary-реплики"
  type        = string
  default     = "staging"
}

# --- Веса для blue/green промоушена ---
# Меняются через terraform apply -var='weight_production=0' -var='weight_canary=100'
# Это и есть «переключение трафика на новую модель» из задания.
variable "weight_production" {
  description = "Вес production target group (0–999)"
  type        = number
  default     = 100
}

variable "weight_canary" {
  description = "Вес canary target group (0–999)"
  type        = number
  default     = 0
}

variable "acm_certificate_arn" {
  description = "ARN TLS-сертификата для ALB"
  type        = string
}
