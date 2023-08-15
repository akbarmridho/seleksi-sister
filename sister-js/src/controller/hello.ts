import { ContentType, type RequestHandler } from '../http/types'

export const helloWorld: RequestHandler = (request, response) => {
  const accept = new Set((request.headers.get('accept') ?? ContentType.text).split(','))

  if (accept.has(ContentType.json)) {
    response.sendJson({ message: 'Hello world' })
  } else {
    response.sendText('Hello world')
  }
}
