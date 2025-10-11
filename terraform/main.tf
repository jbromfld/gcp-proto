terraform {
  required_version = ">= 1.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  # Remote state backend (optional - uncomment after setup-gcp.sh creates the bucket)
  # backend "gcs" {
  #   bucket = "YOUR-PROJECT-ID-terraform-state"
  #   prefix = "rag-system/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = "rag-system"
  description   = "Docker repository for RAG system containers"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# Service account for Cloud Run services
resource "google_service_account" "rag_service" {
  account_id   = "rag-service"
  display_name = "RAG System Service Account"
  description  = "Service account for RAG API and UI"
}

# Grant permissions to service account
resource "google_project_iam_member" "rag_service_permissions" {
  for_each = toset([
    "roles/aiplatform.user",              # Vertex AI access
    "roles/secretmanager.secretAccessor", # Secret Manager access
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.rag_service.email}"
}

# Secret for Elasticsearch credentials
resource "google_secret_manager_secret" "elasticsearch_url" {
  secret_id = "elasticsearch-url"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "elasticsearch_password" {
  secret_id = "elasticsearch-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.required_apis]
}

# VPC Connector for Cloud Run to access Elasticsearch (if using GKE)
resource "google_vpc_access_connector" "connector" {
  count         = var.use_vpc_connector ? 1 : 0
  name          = "rag-vpc-connector"
  region        = var.region
  ip_cidr_range = "10.8.0.0/28"
  network       = "default"

  depends_on = [google_project_service.required_apis]
}

# Cloud Scheduler for ETL job
resource "google_cloud_scheduler_job" "etl_trigger" {
  name        = "rag-etl-trigger"
  description = "Trigger RAG ETL pipeline"
  schedule    = "0 2 * * *" # Daily at 2 AM
  time_zone   = "America/New_York"
  region      = var.region

  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_service.etl.status[0].url}/trigger"

    oidc_token {
      service_account_email = google_service_account.rag_service.email
    }
  }

  depends_on = [google_project_service.required_apis]
}

