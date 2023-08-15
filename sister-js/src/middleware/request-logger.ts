import { type MiddlewareHandler } from '../http/types'
import { jsonToString } from '../json/stringifier'

export const logger: MiddlewareHandler = (request, response, next) => {
  console.log('==========BEGIN REQUEST LOGGER==========')
  console.log('HTTP Head')
  console.log(`${request.method} ${request.uri}`)
  console.log('Query param')
  console.log(`${jsonToString(request.query)}`)
  console.log('Headers')

  request.headers.forEach((value, key) => {
    console.log(`${key}:${value}`)
  })
  console.log('==========END REQUEST LOGGER==========')
  next()
}
