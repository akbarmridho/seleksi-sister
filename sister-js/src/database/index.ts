import { Database } from 'sqlite3'
import { open } from 'sqlite'
import * as sqlite3 from 'sqlite3'

let _db: null | Awaited<ReturnType<typeof open>> = null
export const getDb = async () => {
  if (_db === null) {
    _db = await open({
      filename: 'data/database.sqlite',
      driver: Database,
      mode: sqlite3.OPEN_READWRITE | sqlite3.OPEN_CREATE
    })
    await _db.run('PRAGMA journal_mode = WAL;')
  }

  return _db
}
