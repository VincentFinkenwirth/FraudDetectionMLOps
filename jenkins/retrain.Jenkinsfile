// Retraining and deployment. Started by fraud-monitor (schedule, drift or
// performance) or by hand with REASON=manual. Every stage calls one sub-command
// of pipeline/retrain.py; the files in run/ are passed from stage to stage.
// Jenkins loads this file and the code from the release repository (the newest
// commit that passed fraud-ci), so only tested code trains and deploys models.
pipeline {
    agent any
    options {
        disableConcurrentBuilds()     // never two deployments at the same time
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '100'))
    }

    stages {
        stage('Prepare') {
            steps {
                sh 'rm -rf "$WORKSPACE/run" && mkdir -p "$WORKSPACE/run"'
                // The data is not in git; the history sample is read from data/splits/.
                sh 'ln -sfn /project/data data'
                script {
                    currentBuild.description = "code ${env.GIT_COMMIT.take(7)}"
                }
            }
        }
        stage('Snapshot') {
            steps {
                script {
                    // Exit code 3 = not enough labelled data yet. That is not an error.
                    def code = sh(returnStatus: true,
                        script: 'python -m pipeline.retrain snapshot --run-dir "$WORKSPACE/run" --as-of "$AS_OF"')
                    if (code == 3) {
                        env.SKIP_TRAINING = 'true'
                        currentBuild.description = 'skipped: not enough labelled data'
                    } else if (code != 0) {
                        error("Snapshot failed with exit code ${code}")
                    }
                }
            }
        }
        stage('Train') {
            when { expression { env.SKIP_TRAINING != 'true' } }
            steps {
                sh 'python -m pipeline.retrain train --run-dir "$WORKSPACE/run" --reason "$REASON"'
            }
        }
        stage('Evaluate') {
            when { expression { env.SKIP_TRAINING != 'true' } }
            steps {
                sh 'python -m pipeline.retrain evaluate --run-dir "$WORKSPACE/run"'
            }
        }
        stage('Register') {
            when { expression { env.SKIP_TRAINING != 'true' } }
            steps {
                sh 'python -m pipeline.retrain register --run-dir "$WORKSPACE/run"'
            }
        }
        stage('Deploy + smoke test') {
            when { expression { env.SKIP_TRAINING != 'true' } }
            steps {
                // Switches models/current.json, reloads the API and checks it.
                // If the check fails, the script rolls back to the previous version.
                sh 'python -m pipeline.retrain deploy --run-dir "$WORKSPACE/run"'
                script {
                    currentBuild.description = readFile('run/summary.txt').trim()
                }
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'run/*.json, run/summary.txt, run/candidate/*.json',
                             allowEmptyArchive: true
        }
    }
}
