export enum HTTPMethod {
  GET = 'GET',
  POST = 'POST',
  PUT = 'PUT',
  DELETE = 'DELETE'
}

export type HTTPHeaders = Map<string, string>

export type QueryParam = Record<string, string>
