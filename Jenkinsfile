// Lip-sync model fine-tuning pipeline for Jenkins.
// Runs inside a container with Python + CUDA (or CPU). Configure the image and paths for your cluster.

pipeline {
    agent none

    parameters {
        string(name: 'DATA_DIR', defaultValue: 'data/AVLips1 2', description: 'Path to training data (0_real/, 1_fake/)')
        string(name: 'PRETRAINED', defaultValue: 'weights/best_model.pth', description: 'Path to pretrained checkpoint')
        string(name: 'OUTPUT_DIR', defaultValue: 'weights', description: 'Where to save checkpoints')
        string(name: 'EPOCHS', defaultValue: '36', description: 'Total epochs')
        string(name: 'FREEZE_EPOCHS', defaultValue: '8', description: 'Epochs with frozen encoders')
        string(name: 'BATCH_SIZE', defaultValue: '8', description: 'Batch size')
        string(name: 'LOG_EVERY', defaultValue: '5', description: 'Log every N batches (0 = epoch only)')
        string(name: 'VENV_DIR', defaultValue: '', description: 'Path to venv to activate before training (empty = use $WORKSPACE/venv if present)')
        choice(name: 'DEVICE', choices: ['', 'cuda', 'mps', 'cpu'], description: 'Device (empty = auto)')
    }

    stages {
        stage('Fine-tune') {
            agent {
                label 'docker'  // use a label that has Docker or your GPU agent
            }
            options {
                timeout(time: 24, unit: 'HOURS')
            }
            environment {
                DATA_DIR = "${params.DATA_DIR}"
                PRETRAINED = "${params.PRETRAINED}"
                OUTPUT_DIR = "${params.OUTPUT_DIR}"
                EPOCHS = "${params.EPOCHS}"
                FREEZE_EPOCHS = "${params.FREEZE_EPOCHS}"
                BATCH_SIZE = "${params.BATCH_SIZE}"
                LOG_EVERY = "${params.LOG_EVERY}"
                VENV_DIR = "${params.VENV_DIR}"
                LR = '2e-4'
                LR_ENCODER = '2e-5'
                CONTRASTIVE_WEIGHT = '0.1'
                EARLY_STOPPING_PATIENCE = '8'
                DEVICE = "${params.DEVICE}"
            }
            steps {
                sh '''
                    set -e
                    cd "$WORKSPACE"
                    chmod +x scripts/run_finetune_jenkins.sh
                    # Script activates venv first if present (venv/ in repo root or VENV_DIR)
                    ./scripts/run_finetune_jenkins.sh
                '''
            }
            post {
                success {
                    archiveArtifacts artifacts: "${params.OUTPUT_DIR}/*.pth", allowEmptyArchive: true
                    echo "Training finished. Artifacts (${params.OUTPUT_DIR}/*.pth) archived."
                }
                failure {
                    echo "Training failed. Check console log."
                }
            }
        }
    }
}
