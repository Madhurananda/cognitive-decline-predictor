FROM node:18-alpine AS builder

WORKDIR /app

# Copy dependencies and build
COPY package.json package-lock.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Production server
FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["npm", "start"]
