#!/bin/bash
GCLOUD_CONFIG_NAME="default"
PROJECT_NAME="Gemini API"
# Function to check if the user is logged in
check_login() {
  # Check if a user is logged in
  if ! gcloud auth print-access-token > /dev/null 2>&1; then
    echo "You are not logged in. Running 'gcloud auth login'..."
    gcloud auth login
  else
    echo "You are already logged in."
  fi
}

set_gcloud_config() {
  if ! gcloud config configurations list --filter="name=$GCLOUD_CONFIG_NAME" | grep -q "$GCLOUD_CONFIG_NAME"; then
    echo "Creating gcloud configuration: $GCLOUD_CONFIG_NAME"
    gcloud config configurations create "$GCLOUD_CONFIG_NAME"
  else
    echo "gcloud configuration '$GCLOUD_CONFIG_NAME' already exists"
    gcloud config configurations activate "$GCLOUD_CONFIG_NAME"
  fi

  # Set the project for the configuration
  gcloud config set project "$PROJECT_NAME"
  echo "Set project to $PROJECT_NAME in configuration $GCLOUD_CONFIG_NAME"
}

connect_to_service_account() {
  # Set the path to the key file
  KEY_FILE=".google/service_account.json"

  # Check if the key file exists
  if [ ! -f $KEY_FILE ]; then
    echo "Error: Key file not found"
    return 1
  fi

  # Set the environment variable
  export GOOGLE_APPLICATION_CREDENTIALS=$KEY_FILE

  # Activate the service account
  if ! gcloud auth activate-service-account --key-file=$KEY_FILE; then
    echo "Error: Failed to activate service account"
    return 1
  fi
}


check_login
set_gcloud_config
connect_to_service_account