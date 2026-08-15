# syntax=docker/dockerfile:1
# Build the static SPA bundle.
FROM node:22-alpine AS build
WORKDIR /app
# Use the packageManager-pinned pnpm via Corepack.
RUN corepack enable
# The dashboard is self-contained, so the build context is web/.
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/root/.local/share/pnpm/store/v3,sharing=locked \
    pnpm install --frozen-lockfile
# Copy the source, then build the dashboard.
COPY . .
# Serialize the deploy-time player directory (public/player-directory.json,
# written by deploy.py) into a content-hashed MiniSearch payload + discovery
# manifest under public/, which Vite then copies into dist/. Fails the build
# when the artifact is missing and consumes the raw input so it never ships.
RUN node scripts/build-player-index.mjs
RUN --mount=type=cache,id=pnpm-vite,target=/app/node_modules/.vite,sharing=locked \
    pnpm build

# Serve the SPA and proxy only allowlisted Bento routes.
FROM nginx:alpine
# worker_processes is main-context; patch and verify the distro config.
RUN sed -i 's/^worker_processes.*/worker_processes  4;/' /etc/nginx/nginx.conf \
    && grep -q '^worker_processes  4;' /etc/nginx/nginx.conf
COPY --from=build /app/dist /usr/share/nginx/html
# Official-image startup template: the entrypoint renders /etc/nginx/templates
# via envsubst at container start. Substitution is allowlisted to
# BENTO_API_KEY and BENTO_API_URL so native nginx $variables in the template
# are preserved verbatim.
ENV NGINX_ENVSUBST_FILTER='^(BENTO_API_KEY|BENTO_API_URL)$'
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
# Fail container start when the runtime API key is missing; otherwise the
# rendered auth comparison would be against an empty key.
COPY entrypoint.sh /usr/local/bin/nginx-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/nginx-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
EXPOSE 80
# busybox wget ships in nginx:alpine; matches the compose healthcheck.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
