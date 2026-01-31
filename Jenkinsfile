pipeline {
    agent any

    tools {
        maven 'Maven_3'
    }

    stages {

        stage('Build') {
            when {
                branch 'main'
            }
            steps {
                echo 'Running full CI pipeline on main branch'
                bat 'mvn clean package'
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
                bat 'mvn test'
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
