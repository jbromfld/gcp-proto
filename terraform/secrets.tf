# Secret versions (only created if using Elastic Cloud, not GCE)
resource "google_secret_manager_secret_version" "elasticsearch_url" {
  count  = var.use_gce_elasticsearch ? 0 : 1
  secret = google_secret_manager_secret.elasticsearch_url[0].id

  secret_data = var.elasticsearch_url

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_version" "elasticsearch_password" {
  count  = var.use_gce_elasticsearch ? 0 : 1
  secret = google_secret_manager_secret.elasticsearch_password[0].id

  secret_data = var.elasticsearch_password

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# Grant service account access to secrets (only if using Elastic Cloud)
resource "google_secret_manager_secret_iam_member" "elasticsearch_url_access" {
  count     = var.use_gce_elasticsearch ? 0 : 1
  secret_id = google_secret_manager_secret.elasticsearch_url[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_service.email}"
}

resource "google_secret_manager_secret_iam_member" "elasticsearch_password_access" {
  count     = var.use_gce_elasticsearch ? 0 : 1
  secret_id = google_secret_manager_secret.elasticsearch_password[0].id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.rag_service.email}"
}

