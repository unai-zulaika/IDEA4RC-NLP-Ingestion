# --- Stage 1: Build the Vite + React app ---
FROM node:18-alpine AS builder
WORKDIR /app

# Copy only package manifests
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of your source
COPY . .

RUN apk add --no-cache bash

ARG BASE_URL_PATH=/
ENV BASE_URL_PATH=$BASE_URL_PATH

# Build the app
RUN npm run build

# --- Stage 2: Serve with NGINX ---
FROM nginx:alpine
RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
