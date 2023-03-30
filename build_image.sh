USER_UID=$(id -u)
USER_GID=$(getent group docker | cut -d: -f3)
docker build --build-arg USER_UID=$USER_UID --build-arg USER_GID=$USER_GID --build-arg USERNAME=minhbq6 . --tag deepstream-app
# docker buildx build --build-arg USER_UID=$USER_UID --build-arg USER_GID=$USER_GID --runtime=nvidia . --tag deepstream-app
