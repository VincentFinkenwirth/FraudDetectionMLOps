# Jenkins with Python and all project libraries, so the pipeline stages can run
# the same Python scripts I use locally. The jobs take the code from git (see
# casc.yaml): fraud-ci tests every new commit and only releases green commits,
# fraud-monitor and fraud-retrain run the released code.
FROM jenkins/jenkins:lts-jdk21

USER root
# The Debian of this image ships Python 3.13, but the API image uses 3.12. A
# model should be trained and served with the same Python and library
# versions, so uv installs Python 3.12 and the packages from the lock file.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.12 && uv venv --seed --python 3.12 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY requirements.lock /tmp/requirements.lock
RUN uv pip install --python /opt/venv/bin/python --no-cache -r /tmp/requirements.lock

# Jenkins clones the mounted repository (/repo.git) and its own release
# repository. Git refuses repositories that belong to another user unless they
# are marked as safe, and the bind mount belongs to root.
RUN git config --system --add safe.directory '*'

# Plugins + "configuration as code": users and jobs are created at start-up,
# so nobody has to click through the setup wizard. ALLOW_LOCAL_CHECKOUT lets
# the git plugin clone file:// repositories (blocked by default since 4.11).
COPY jenkins/plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
COPY jenkins/casc.yaml /usr/share/jenkins/ref/casc.yaml
ENV CASC_JENKINS_CONFIG=/usr/share/jenkins/ref/casc.yaml \
    JAVA_OPTS="-Djenkins.install.runSetupWizard=false -Dhudson.plugins.git.GitSCM.ALLOW_LOCAL_CHECKOUT=true" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER jenkins
