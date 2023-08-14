import { availableParallelism } from 'os'
import { errorHandler } from './http/handler/internal-error'
import { notFoundHandler } from './http/handler/not-found'
import { Http } from './http/server'
import cluster from 'cluster'

async function startServer () {
  const PORT = 3000

  const server = new Http()

  server.useErrorMiddleware(notFoundHandler)
  server.useErrorMiddleware(errorHandler)

  server.serve(PORT)
}

const availableCpu = availableParallelism()

if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} is running`)

  // Fork workers.
  for (let i = 0; i < availableCpu; i++) {
    cluster.fork()
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`worker ${worker.process.pid ?? ''} died`)
  })
} else {
  startServer().catch(console.log)
  console.log(`Worker ${process.pid} started`)
}
