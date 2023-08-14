import { errorHandler } from './http/handler/internal-error'
import { notFoundHandler } from './http/handler/not-found'
import { Http } from './http/server'

(async () => {
  const PORT = 3000

  const server = new Http()

  server.useErrorMiddleware(notFoundHandler)
  server.useErrorMiddleware(errorHandler)

  server.serve(PORT)
})().catch(e => { console.log(e) })
