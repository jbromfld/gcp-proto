variable "project_id" {
  description = "GCP Project ID"
  type        = string
  # Set via: export TF_VAR_project_id=YOUR-PROJECT-ID
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "elasticsearch_url" {
  description = "Elasticsearch URL (Elastic Cloud or GKE service)"
  type        = string
  # Example: "https://your-deployment.es.us-central1.gcp.cloud.es.io:9243"
}

variable "elasticsearch_password" {
  description = "Elasticsearch password"
  type        = string
  sensitive   = true
}

variable "use_vpc_connector" {
  description = "Whether to create VPC connector (needed for GKE Elasticsearch)"
  type        = bool
  default     = false
}

variable "scrape_urls" {
  description = "Comma-separated URLs to scrape"
  type        = string
  default     = "https://docs.python.org/3/tutorial/,https://fastapi.tiangolo.com/"
}

variable "api_cpu" {
  description = "CPU allocation for API service"
  type        = string
  default     = "2"
}

variable "api_memory" {
  description = "Memory allocation for API service"
  type        = string
  default     = "4Gi"
}

variable "api_min_instances" {
  description = "Minimum instances for API"
  type        = number
  default     = 0
}

variable "api_max_instances" {
  description = "Maximum instances for API"
  type        = number
  default     = 10
}

variable "allow_unauthenticated" {
  description = "Allow unauthenticated access to services"
  type        = bool
  default     = true # Set to false for production with IAP/authentication
}

