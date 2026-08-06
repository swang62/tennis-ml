# Build stage: install deps and produce the static SPA bundle.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Runtime stage: serve the built SPA with nginx; /api is proxied to the Bento.
FROM nginx:alpine
# One nginx worker: the static SPA needs no concurrency, and the image default
# `worker_processes auto` would start one worker per visible CPU (10 here).
# worker_processes is main-context, so it cannot live in conf.d/default.conf;
# patch the distro main config and fail the build loudly if the line ever
# changes shape upstream. The grep assertion also catches a removed directive.
RUN sed -i 's/^worker_processes.*/worker_processes  1;/' /etc/nginx/nginx.conf \
    && grep -q '^worker_processes  1;' /etc/nginx/nginx.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
