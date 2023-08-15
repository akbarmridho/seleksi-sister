import { availableParallelism } from 'os'
import { Http } from './http/server'
import cluster from 'cluster'
import { helloWorld } from './controller/hello'
import { notFoundHandler } from './middleware/not-found'
import { errorHandler } from './middleware/internal-error'
import { logger } from './middleware/request-logger'

async function startServer () {
  const PORT = 3000

  const server = new Http()

  server.useMiddleware(logger)

  server.get('/', helloWorld)

  server.useErrorMiddleware(notFoundHandler)
  server.useErrorMiddleware(errorHandler)

  server.serve(PORT)
}

const MAX_WORKER = 1

const availableCpu = availableParallelism()
console.log(`Available CPU: ${availableCpu}, MAX: ${MAX_WORKER}`)

if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} is running`)
  const workerNumber = MAX_WORKER < availableCpu ? MAX_WORKER : availableCpu

  // Fork workers.
  for (let i = 0; i < workerNumber; i++) {
    cluster.fork()
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`worker ${worker.process.pid ?? ''} died`)
  })
} else {
  startServer().catch(console.log)
  console.log(`Worker ${process.pid} started`)
}
