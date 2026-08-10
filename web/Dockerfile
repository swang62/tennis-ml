# Build the static SPA bundle.
FROM node:22-alpine AS build
WORKDIR /app
# Use the packageManager-pinned pnpm via Corepack.
RUN corepack enable
# The dashboard is self-contained, so the build context is web/.
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
# Copy the source, then build the dashboard.
COPY . .
RUN pnpm build

# Serve the SPA and proxy only allowlisted Bento routes.
FROM nginx:alpine
# worker_processes is main-context; patch and verify the distro config.
RUN sed -i 's/^worker_processes.*/worker_processes  1;/' /etc/nginx/nginx.conf \
    && grep -q '^worker_processes  1;' /etc/nginx/nginx.conf
COPY --from=build /app/dist /usr/share/nginx/html
# Official-image startup template: the entrypoint renders /etc/nginx/templates
# via envsubst at container start. Substitution is allowlisted to
# DRIFT_API_KEY and BENTO_API_URL so native nginx $variables in the template
# are preserved verbatim.
ENV NGINX_ENVSUBST_FILTER='^(DRIFT_API_KEY|BENTO_API_URL)$'
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
