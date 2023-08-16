import { ContentType, type RequestHandler } from '../http/types'
import { getDb } from '../database'
import { BadRequestException } from '../http/exception'
import * as fs from 'fs/promises'

interface Todo {
  id: number
  title: string
  file: string | null
  completed: boolean
}
export const getAllTodo: RequestHandler = async (request, response) => {
  const isCompleted = 'completed' in request.query ? (request.query.completed === 'true') : undefined
  const db = await getDb()

  let result: Todo[]

  if (isCompleted !== undefined) {
    result = await db.all<Todo[]>('SELECT * FROM todo where completed = ?', isCompleted)
  } else {
    result = await db.all<Todo[]>('SELECT * FROM todo')
  }

  response.sendJson(result)
}

export const addTodo: RequestHandler = async (request, response) => {
  const form = request.formMultipart()
  /**
   * Form data format
   * title: string
   * file: jpg
   */

  form.forEach(val => {
    console.log(val.name)
    console.log(val.contentType)
  })

  const title = form.get('title')
  const file = form.get('file')

  if (title === undefined || file === undefined) {
    throw new BadRequestException('Title and file is required')
  }

  if (file.contentType !== ContentType.png || file.filename === undefined) {
    throw new BadRequestException('Image must be png and have filename')
  }

  await fs.writeFile(`storage/${file.filename}`, file.data)
  const db = await getDb()

  await db.run('INSERT INTO todo(title, file) VALUES (?, ?)', title.data.toString(), file.filename)

  response.sendJson({ message: 'Created' })
}

export const setComplete: RequestHandler = async (request, response) => {
  const form = request.json()

  if (!('id' in form)) {
    throw new BadRequestException('Id is required')
  }

  const db = await getDb()

  await db.run('UPDATE todo SET completed = true WHERE id = ?', form.id)

  response.sendJson({ message: 'Updated' })
}

export const deleteTodo: RequestHandler = async (request, response) => {
  const form = request.urlEncoded()

  if (!('id' in form)) {
    throw new BadRequestException('Id is required')
  }

  const db = await getDb()

  await db.run('DELETE from todo WHERE id = ?', form.id)

  response.sendJson({ message: 'Deleted' })
}
