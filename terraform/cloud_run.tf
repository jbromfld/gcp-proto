# Cloud Run service for API
resource "google_cloud_run_service" "api" {
  name     = "rag-api"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.rag_service.email

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/rag-api:latest"

        resources {
          limits = {
            cpu    = var.api_cpu
            memory = var.api_memory
          }
        }

        env {
          name = "ELASTICSEARCH_URL"
          value = var.use_gce_elasticsearch ? "http://${google_compute_instance.elasticsearch[0].network_interface[0].network_ip}:9200" : var.elasticsearch_url
        }

        env {
          name = "ELASTICSEARCH_PASSWORD"
          value = var.use_gce_elasticsearch ? "" : var.elasticsearch_password
        }

        env {
          name  = "EMBEDDING_PROVIDER"
          value = "vertex"
        }

        env {
          name  = "LLM_PROVIDER"
          value = "vertex"
        }

        env {
          name  = "GOOGLE_PROJECT_ID"
          value = var.project_id
        }

        env {
          name  = "GOOGLE_REGION"
          value = var.region
        }

        ports {
          container_port = 8000
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"        = var.api_min_instances
        "autoscaling.knative.dev/maxScale"        = var.api_max_instances
        "run.googleapis.com/vpc-access-connector" = var.use_vpc_connector ? google_vpc_access_connector.connector[0].id : null
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.required_apis,
    time_sleep.wait_for_elasticsearch,
  ]
}

# Cloud Run service for UI
resource "google_cloud_run_service" "ui" {
  name     = "rag-ui"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.rag_service.email

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/rag-ui:latest"

        resources {
          limits = {
            cpu    = "1"
            memory = "2Gi"
          }
        }

        env {
          name  = "API_URL"
          value = google_cloud_run_service.api.status[0].url
        }

        ports {
          container_port = 8501
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "0"
        "autoscaling.knative.dev/maxScale" = "5"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_cloud_run_service.api,
    time_sleep.wait_for_elasticsearch,
  ]
}

# Cloud Run service for ETL
resource "google_cloud_run_service" "etl" {
  name     = "rag-etl"
  location = var.region

  template {
    spec {
      service_account_name = google_service_account.rag_service.email
      timeout_seconds      = 3600 # 1 hour for ETL jobs

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/rag-etl:latest"

        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }

        env {
          name = "ELASTICSEARCH_URL"
          value = var.use_gce_elasticsearch ? "http://${google_compute_instance.elasticsearch[0].network_interface[0].network_ip}:9200" : var.elasticsearch_url
        }

        env {
          name = "ELASTICSEARCH_PASSWORD"
          value = var.use_gce_elasticsearch ? "" : var.elasticsearch_password
        }

        env {
          name  = "EMBEDDING_PROVIDER"
          value = "vertex"
        }

        env {
          name  = "GOOGLE_PROJECT_ID"
          value = var.project_id
        }

        ports {
          container_port = 8080
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale"        = "0"
        "autoscaling.knative.dev/maxScale"        = "1"
        "run.googleapis.com/vpc-access-connector" = var.use_vpc_connector ? google_vpc_access_connector.connector[0].id : null
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  depends_on = [
    google_project_service.required_apis,
    time_sleep.wait_for_elasticsearch,
  ]
}

# IAM policy to allow unauthenticated access (optional - for demo)
resource "google_cloud_run_service_iam_member" "api_public" {
  count    = var.allow_unauthenticated ? 1 : 0
  service  = google_cloud_run_service.api.name
  location = google_cloud_run_service.api.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_service_iam_member" "ui_public" {
  count    = var.allow_unauthenticated ? 1 : 0
  service  = google_cloud_run_service.ui.name
  location = google_cloud_run_service.ui.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Grant Cloud Scheduler permission to invoke ETL
resource "google_cloud_run_service_iam_member" "etl_scheduler" {
  service  = google_cloud_run_service.etl.name
  location = google_cloud_run_service.etl.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.rag_service.email}"
}

