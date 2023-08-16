import { availableParallelism } from 'os'
import { Http } from './http/server'
import cluster from 'cluster'
import { helloWorld } from './controller/hello'
import { notFoundHandler } from './middleware/not-found'
import { errorHandler } from './middleware/internal-error'
import { logger } from './middleware/request-logger'
import { badRequestHandler } from './middleware/bad-request'
import { getDb } from './database'
import { addTodo, deleteTodo, getAllTodo, setComplete } from './controller/todo'
import { getFile } from './controller/file'

async function startServer () {
  const PORT = 3000
  const db = await getDb()

  // init db
  await db.exec(`create table if not exists todo
                (
                    id integer primary key autoincrement,
                    title text not null,
                    file text,
                    completed boolean not null default false
                )`)

  const server = new Http()

  server.useMiddleware(logger)

  server.get('/', helloWorld)

  server.get('/todo', getAllTodo)
  server.post('/todo', addTodo)
  server.put('/todo', setComplete)
  server.delete('/todo', deleteTodo)
  server.get('/file', getFile)

  server.useErrorMiddleware(badRequestHandler)
  server.useErrorMiddleware(notFoundHandler)
  server.useErrorMiddleware(errorHandler)

  server.serve(PORT)
}

const MAX_WORKER = 2

const availableCpu = availableParallelism()
console.log(`Available CPU: ${availableCpu}, MAX: ${MAX_WORKER}`)

if (cluster.isPrimary) {
  console.log(`Primary ${process.pid} is running`)
  const workerNumber = MAX_WORKER < availableCpu ? MAX_WORKER : availableCpu
  getDb().catch(console.log)

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
