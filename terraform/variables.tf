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

# ============================================
# Elasticsearch on GCE (Cost-effective option)
# ============================================

variable "use_gce_elasticsearch" {
  description = "Deploy Elasticsearch on GCE instead of using Elastic Cloud"
  type        = bool
  default     = true # Set to false if using Elastic Cloud
}

variable "elasticsearch_machine_type" {
  description = "Machine type for Elasticsearch VM"
  type        = string
  default     = "e2-medium" # 2 vCPU, 4GB RAM - ~$25/mo
  # Options: e2-small ($12/mo), e2-medium ($25/mo), e2-standard-2 ($50/mo)
}

variable "elasticsearch_disk_size_gb" {
  description = "Size of persistent disk for Elasticsearch data (GB)"
  type        = number
  default     = 50 # ~$5/mo for pd-balanced
}

variable "elasticsearch_heap_size" {
  description = "Elasticsearch JVM heap size (should be ~50% of RAM)"
  type        = string
  default     = "2g" # For e2-medium with 4GB RAM
}

variable "elasticsearch_use_preemptible" {
  description = "Use preemptible VM for Elasticsearch (cheaper but can be terminated)"
  type        = bool
  default     = false # Set to true for dev/testing to save 60-80%
}

