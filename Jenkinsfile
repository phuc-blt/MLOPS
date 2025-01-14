pipeline {
    agent any

    triggers {
        GenericTrigger(
            genericVariables: [
                [key: 'WEBHOOK_TRIGGER', value: '$.trigger', defaultValue: '']
            ],
            causeString: 'Triggered by webhook',
            token: 'phuc',
            printContributedVariables: true,
            printPostContent: true
        )
    }

    options {
        skipDefaultCheckout() // Prevent automatic checkout
    }

    stages {
        stage('Start Pipeline') {
            steps {
                withChecks('Start Pipeline') {
                    publishChecks name: 'Start Pipeline', status: 'IN_PROGRESS', summary: 'Pipeline execution has started.'
                }
            }
        }

        stage('Clone Repository') {
            steps {
                git branch: "main", url: 'https://github.com/phuc-blt/MLOPS.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                dir('docker-image') { // Move into the docker-image directory
                    script {
                        try {
                            sh '''
                            #!/bin/bash

                            # Remove existing Docker image
                            if sudo docker images | grep -q "api"; then
                                echo "Removing existing Docker image..."
                                sudo docker rmi -f api
                            fi

                            # Build the Docker image
                            echo "Building the Docker image..."
                            sudo docker build -t app-api .
                            '''

                            withChecks('Build Docker Image') {
                                publishChecks name: 'Build Docker Image', status: 'COMPLETED', conclusion: 'SUCCESS',
                                             summary: 'Docker image built successfully.'
                            }
                        } catch (e) {
                            withChecks('Build Docker Image') {
                                publishChecks name: 'Build Docker Image', status: 'COMPLETED', conclusion: 'FAILURE',
                                             summary: 'Pipeline failed while building the Docker image.'
                            }
                            throw e
                        }
                    }
                }
            }
        }

        stage('Run Docker Container') {
            steps {
                script {
                    try {
                        sh '''
                        #!/bin/bash

                        # Stop and remove any existing container
                        if sudo docker ps -a --format '{{.Names}}' | grep -q "^api_running$"; then
                            echo "Container 'api_running' already exists. Removing it..."
                            sudo docker stop api_running
                            sudo docker rm -f api_running
                        fi

                        # Run the Docker container
                        echo "Running the Docker container..."
                        sudo docker run app-api
                        '''

                        withChecks('Run Docker Container') {
                            publishChecks name: 'Run Docker Container', status: 'COMPLETED', conclusion: 'SUCCESS',
                                         summary: 'Docker container running successfully.'
                        }
                    } catch (e) {
                        withChecks('Run Docker Container') {
                            publishChecks name: 'Run Docker Container', status: 'COMPLETED', conclusion: 'FAILURE',
                                         summary: 'Pipeline failed while running the Docker container.'
                        }
                        throw e
                    }
                }
            }
        }
    }

    post {
        always {
            withChecks('Pipeline Completion') {
                publishChecks name: 'Pipeline Completion', status: 'COMPLETED', conclusion: 'NEUTRAL',
                             summary: 'Pipeline has completed execution.'
            }
        }
    }
}
