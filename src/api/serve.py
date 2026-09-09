### API IMPORTS ###

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import ml.logistic_interface as logistic_interface
import ml.svm_interface as svm_interface
import logging

### API FUNCTIONALITY ###

api = FastAPI()
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s"
)

# GET Health Endpoint:
@api.get('/api/health')
def health():
  try:
    return JSONResponse(content={'status': 'OK'}, status_code=200)

  except Exception as e:
    logging.error('Unexpected Error in GET /api/health: ' + str(e))
    return JSONResponse(content={'status': 'Server Error'}, status_code=500)

# POST Logistic Endpoint:
@api.post('/api/logistic')
async def logistic_classify(request: Request):
  try:
    if not request.is_json:
      logging.error('Malformed Request Body')
      return JSONResponse(content={'status': 'Not Acceptable'}, status_code=406)

    data = await request.json()
    if type(data) is not list:
      logging.error('Request Body Not a List')
      return JSONResponse(content={'status': 'Not Acceptable'}, status_code=406)

    predictions = logistic_interface.classify(data)
    labels = logistic_interface.label_output(predictions, lookup=None)
    return JSONResponse(
      content = {
        'classifications': labels
      },
      status_code = 200
    )

  except Exception as e:
    logging.error('Unexpected Error in POST /api/logistic: ' + str(e))
    return JSONResponse(content={'status': 'Server Error'}, status_code=500)

# POST SVM Endpoint:
@api.post('/api/svm')
async def svm_classify(request: Request):
  try:
    if not request.is_json:
      logging.error('Malformed Request Body')
      return JSONResponse(content={'status': 'Not Acceptable'}, status_code=406)

    data = await request.json()
    if type(data) is not list:
      logging.error('Request Body Not a List')
      return JSONResponse(content={'status': 'Not Acceptable'}, status_code=406)

    predictions = svm_interface.classify(data)
    labels = svm_interface.label_output(predictions, lookup=None)
    return JSONResponse(
      content = {
        'classifications': labels
      },
      status_code = 200
    )

  except Exception as e:
    logging.error('Unexpected Error in POST /api/svm: ' + str(e))
    return JSONResponse(content={'status': 'Server Error'}, status_code=500)