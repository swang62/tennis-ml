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
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
