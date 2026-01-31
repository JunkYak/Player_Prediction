pipeline {
    agent any

    stages {

        stage('Build') {
            when {
                branch 'main'
            }
            steps {
                echo 'Running full CI pipeline on main branch'
                sh 'mvn clean package'
            }
        }

        stage('Test') {
            when {
                anyOf {
                    branch 'main'
                    branch 'feature/*'
                }
            }
            steps {
                echo 'Running tests'
                sh 'mvn test'
            }
        }

        stage('Security Scan') {
            when {
                branch 'release/*'
            }
            steps {
                echo 'Running security scan on release branch'
            }
        }
    }
}
