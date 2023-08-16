import { ContentType } from '../types'

export interface FormValue {
  contentType: string
  name: string
  filename?: string
  data: Buffer
}

export const parseMultipart = (boundary: string, buffer: Buffer) => {
  const bufferStr = buffer.toString('binary').trim().split(`--${boundary}`).map(e => e.trim())
  const result: FormValue[] = []

  bufferStr.forEach(each => {
    if (each !== '--' && each.length !== 0) {
      const [headers, data] = each.split('\r\n\r\n')
      let contentType: string = ContentType.text
      let filename: string | undefined
      let name = ''

      let rawHeaders: string[] = []

      headers.split('\r\n').forEach(each => {
        rawHeaders = [...rawHeaders, ...each.split(';').map(e => e.trim())]
      })

      rawHeaders.forEach(raw => {
        if (raw.startsWith('name')) {
          const [, value] = raw.split('=')
          name = value.slice(1, -1) // remove double quote
        } else if (raw.toLowerCase().startsWith('content-type')) {
          const [, value] = raw.split(':')
          contentType = value.trim()
        } else if (raw.startsWith('filename')) {
          const [, value] = raw.split('=')
          filename = value.slice(1, -1) // remove double quote
        }
      })

      result.push({
        contentType,
        filename,
        name,
        data: Buffer.from(data, 'binary')
      })
    }
  })

  return result
}
