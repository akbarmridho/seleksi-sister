import { type Server, type Socket, createServer } from 'net'
import { type Route, type MiddlewareHandler, type ErrorHandler, HTTPStatus, type RequestHandler, HTTPMethod } from './types'
import { parseHttpRequest } from './parser'
import { Response } from './response'
import { HandlerNotFound } from './exception'

export class Http {
  private readonly server: Server
  private readonly middlewarePipeline: MiddlewareHandler[]
  private readonly requestPileline: Route[]
  private readonly errorPipeline: ErrorHandler[]

  public constructor () {
    this.middlewarePipeline = []
    this.requestPileline = []
    this.errorPipeline = []

    this.server = createServer(this.requestHandler.bind(this))
  }

  public serve (port: number) {
    this.server.listen(port, () => {
      console.log(`Server opened on  http://localhost:${port}`)
    })
  }

  private requestHandler (socket: Socket) {
    socket.on('data', (data: Buffer) => {
      const request = parseHttpRequest(data)
      const response = new Response(HTTPStatus.OK, socket);

      (async () => {
        let nextCalled: boolean = true

        for (const handler of this.middlewarePipeline) {
          nextCalled = false

          const next = () => {
            nextCalled = true
          }

          await handler(request, response, next)

          if (!nextCalled) {
            break
          }
        }

        if (!nextCalled) {
          return
        }

        const selectedHandler = this.requestPileline.find(hand => (
          hand.endpoint === request.uri &&
          hand.method === request.method
        ))

        if (selectedHandler !== undefined) {
          await selectedHandler.handler(request, response)
        } else {
          throw new HandlerNotFound(`Route ${request.method} ${request.uri} does not have corresponding handler`)
        }
      })().catch(e => {
        if (e instanceof Error) {
          let nextCalled = true

          for (const handler of this.errorPipeline) {
            nextCalled = false

            const next = () => {
              nextCalled = true
            }

            handler(request, response, next, e)

            if (!nextCalled) {
              break
            }
          }

          if (!nextCalled) {
            throw e
          }
        } else {
          throw e
        }
      })
    })
  }

  private addRoute (method: HTTPMethod, endpoint: string, handler: RequestHandler) {
    this.requestPileline.push({
      method,
      endpoint,
      handler
    })
  }

  public get (route: string, handler: RequestHandler) {
    this.addRoute(HTTPMethod.GET, route, handler)
  }

  public post (route: string, handler: RequestHandler) {
    this.addRoute(HTTPMethod.POST, route, handler)
  }

  public put (route: string, handler: RequestHandler) {
    this.addRoute(HTTPMethod.PUT, route, handler)
  }

  public delete (route: string, handler: RequestHandler) {
    this.addRoute(HTTPMethod.DELETE, route, handler)
  }

  public useMiddleware (handler: MiddlewareHandler) {
    this.middlewarePipeline.push(handler)
  }

  public useErrorMiddleware (handler: ErrorHandler) {
    this.errorPipeline.push(handler)
  }
}
