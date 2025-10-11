# Self-hosted Elasticsearch on single GCE instance (cost-effective alternative to Elastic Cloud)
# This deploys Elasticsearch on a single VM with persistent disk

# Firewall rule to allow Cloud Run to access Elasticsearch
resource "google_compute_firewall" "elasticsearch" {
  count   = var.use_gce_elasticsearch ? 1 : 0
  name    = "allow-elasticsearch"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["9200", "9300"]
  }

  source_ranges = ["0.0.0.0/0"] # Restrict this in production
  target_tags   = ["elasticsearch"]

  depends_on = [google_project_service.required_apis]
}

# Persistent disk for Elasticsearch data
resource "google_compute_disk" "elasticsearch_data" {
  count = var.use_gce_elasticsearch ? 1 : 0
  name  = "elasticsearch-data"
  type  = "pd-balanced" # or pd-ssd for better performance
  zone  = "${var.region}-a"
  size  = var.elasticsearch_disk_size_gb

  depends_on = [google_project_service.required_apis]
}

# Elasticsearch VM instance
resource "google_compute_instance" "elasticsearch" {
  count        = var.use_gce_elasticsearch ? 1 : 0
  name         = "elasticsearch"
  machine_type = var.elasticsearch_machine_type
  zone         = "${var.region}-a"

  tags = ["elasticsearch"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 20 # OS disk
    }
  }

  attached_disk {
    source      = google_compute_disk.elasticsearch_data[0].id
    device_name = "elasticsearch-data"
  }

  network_interface {
    network = "default"

    access_config {
      # Ephemeral external IP (for admin access)
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e
    
    # Format and mount data disk if needed
    if ! mount | grep -q /var/lib/elasticsearch; then
      mkfs.ext4 -F /dev/disk/by-id/google-elasticsearch-data || true
      mkdir -p /var/lib/elasticsearch
      mount /dev/disk/by-id/google-elasticsearch-data /var/lib/elasticsearch
      echo '/dev/disk/by-id/google-elasticsearch-data /var/lib/elasticsearch ext4 defaults 0 0' >> /etc/fstab
    fi
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
      curl -fsSL https://get.docker.com | sh
      systemctl enable docker
      systemctl start docker
    fi
    
    # Run Elasticsearch container
    docker stop elasticsearch || true
    docker rm elasticsearch || true
    
    docker run -d \
      --name elasticsearch \
      --restart unless-stopped \
      -p 9200:9200 \
      -p 9300:9300 \
      -e "discovery.type=single-node" \
      -e "xpack.security.enabled=false" \
      -e "ES_JAVA_OPTS=-Xms${var.elasticsearch_heap_size} -Xmx${var.elasticsearch_heap_size}" \
      -e "cluster.name=rag-cluster" \
      -e "bootstrap.memory_lock=true" \
      -v /var/lib/elasticsearch:/usr/share/elasticsearch/data \
      --ulimit memlock=-1:-1 \
      docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    
    # Wait for Elasticsearch to start
    sleep 30
    
    # Create health check endpoint
    cat > /usr/local/bin/es-health.sh <<'HEALTH'
    #!/bin/bash
    curl -sf http://localhost:9200/_cluster/health || exit 1
    HEALTH
    chmod +x /usr/local/bin/es-health.sh
    
    echo "Elasticsearch started successfully"
  EOF

  service_account {
    email  = google_service_account.rag_service.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    preemptible       = var.elasticsearch_use_preemptible
    automatic_restart = !var.elasticsearch_use_preemptible
  }

  depends_on = [google_compute_disk.elasticsearch_data]
}

# Output Elasticsearch internal IP
output "elasticsearch_internal_ip" {
  description = "Internal IP of Elasticsearch instance"
  value       = var.use_gce_elasticsearch ? google_compute_instance.elasticsearch[0].network_interface[0].network_ip : null
}

output "elasticsearch_external_ip" {
  description = "External IP of Elasticsearch instance (for admin access)"
  value       = var.use_gce_elasticsearch ? google_compute_instance.elasticsearch[0].network_interface[0].access_config[0].nat_ip : null
}

