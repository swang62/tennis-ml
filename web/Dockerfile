# Build stage: install deps and produce the static SPA bundle.
FROM node:22-alpine AS build
WORKDIR /app
# Enable Corepack (bundled with Node.js) so `pnpm` resolves to the version
# pinned in package.json's `packageManager` field.
RUN corepack enable
# The dashboard is self-contained, so the build context is web/.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
# Copy the source, then build the dashboard.
COPY . .
RUN pnpm build

# Runtime stage: serve the built SPA with nginx; only allowlisted API routes
# are proxied to the Bento.
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
