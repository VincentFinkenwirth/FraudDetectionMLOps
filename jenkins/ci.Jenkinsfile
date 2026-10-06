// CI for the code. Jenkins polls the repository every minute and runs this for
// every new commit on main. Only a green commit is pushed to the release
// repository, and fraud-monitor / fraud-retrain only run released code. So no
// model is ever trained (and no threshold changed) by untested code.
pipeline {
    agent any
    options {
        disableConcurrentBuilds()
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '50'))
    }
    environment {
        RELEASE_REPO = "${JENKINS_HOME}/release.git"
        // The Jenkins container knows the real database, MLflow server and API.
        // The tests must never touch them, so these variables are removed.
        ISOLATED = 'env -u DATABASE_URL -u MLFLOW_TRACKING_URI -u API_URL -u API_KEY_ADMIN -u MODEL_DIR'
    }

    stages {
        stage('Prepare') {
            steps {
                // The data is not in git (213 MB); the smoke test reads a sample of it.
                sh 'ln -sfn /project/data data'
            }
        }
        stage('Environment') {
            steps {
                // Installed packages must be exactly the ones in requirements.lock.
                sh 'python -m pip check && python -m pipeline.check_environment'
            }
        }
        stage('Unit tests') {
            steps {
                sh '$ISOLATED python -m unittest discover -s tests -t .'
            }
        }
        stage('Training smoke test') {
            steps {
                sh '$ISOLATED python -m pipeline.smoke_test --output "$WORKSPACE/smoke"'
            }
        }
        stage('Release') {
            steps {
                // --force: the release always follows main, even after a rewritten history.
                sh '''
                    test -d "$RELEASE_REPO" || git init --bare --initial-branch=main "$RELEASE_REPO"
                    git push --force "$RELEASE_REPO" "$GIT_COMMIT:refs/heads/main"
                '''
                script {
                    currentBuild.description = "released ${env.GIT_COMMIT.take(7)}"
                }
            }
        }
    }
}
