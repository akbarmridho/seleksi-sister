import { ContentType, ContentTypeHeader, type RequestHandler } from '../http/types'
import { BadRequestException } from '../http/exception'
import * as fs from 'fs/promises'

export const getFile: RequestHandler = async (request, response) => {
  if (!('filename' in request.query)) {
    throw new BadRequestException('filename query is required')
  }

  const filename = request.query.filename

  const file = await fs.readFile(`storage/${filename}`)

  response.addHeader(ContentTypeHeader, ContentType.png).send(file, 'binary')
}
