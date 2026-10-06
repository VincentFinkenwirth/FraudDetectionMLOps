// Weekly monitoring job. Jenkins loads this file and the code from the release
// repository, i.e. the newest commit that passed fraud-ci.
// In the simulation, simulation/replay.py starts it once per simulated week and
// passes AS_OF. In real operation the cron trigger below would start it every
// Monday (I left it commented out so it does not interfere with the simulation).
pipeline {
    agent any
    options {
        disableConcurrentBuilds()
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '300'))
    }
    // triggers { cron('H 6 * * 1') }

    stages {
        stage('Monitor') {
            steps {
                // The workspace is a checkout of the released commit; results go to out/.
                sh 'python -m pipeline.monitor --as-of "$AS_OF" --output "$WORKSPACE/out"'
            }
        }
        stage('Decide') {
            steps {
                script {
                    def decision = readFile('out/decision.txt').trim()
                    def reasons = readFile('out/retrain_reason.txt').trim()
                    currentBuild.description = reasons ? "${decision} (${reasons})" : decision

                    if (decision == 'retrain') {
                        def run = build(job: 'fraud-retrain', wait: true, propagate: false, parameters: [
                            string(name: 'AS_OF', value: params.AS_OF),
                            string(name: 'REASON', value: reasons),
                        ])
                        currentBuild.description += " -> ${run.description ?: run.result}"
                        if (run.result != 'SUCCESS') {
                            unstable("The retraining job finished with ${run.result}.")
                        }
                    }
                    if (['blocked_quality', 'quality_incident'].contains(decision)) {
                        // Yellow build = someone has to look at the data source.
                        unstable('Data-quality incident, see out/monitoring_report.json.')
                    }
                }
            }
        }
    }
    post {
        always {
            archiveArtifacts artifacts: 'out/*', allowEmptyArchive: true
        }
    }
}
