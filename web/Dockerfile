# Build the static SPA bundle.
FROM node:24-alpine AS build

WORKDIR /app
ARG VITE_SITE_URL
ARG VITE_SITE_ID
ENV VITE_SITE_URL=$VITE_SITE_URL VITE_SITE_ID=$VITE_SITE_ID

# Use the packageManager-pinned pnpm via Corepack.
RUN corepack enable

# The dashboard is self-contained, so the build context is web/.
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store,sharing=locked \
  pnpm install --frozen-lockfile --store-dir /pnpm/store

COPY . .

# minisearch and player directory index
RUN node scripts/build-player-index.mjs

# Final production build
RUN --mount=type=cache,id=pnpm-vite,target=/app/node_modules/.vite,sharing=locked \
  pnpm build

# Serve the SPA and proxy only allowlisted Bento routes.
FROM nginx:alpine

ENV NGINX_ENVSUBST_FILTER='^(BENTO_API_KEY|BENTO_API_URL)$'
EXPOSE 80

RUN sed -i 's/^worker_processes.*/worker_processes  4;/' /etc/nginx/nginx.conf \
  && grep -q '^worker_processes  4;' /etc/nginx/nginx.conf

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY entrypoint.sh /usr/local/bin/nginx-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/nginx-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]
