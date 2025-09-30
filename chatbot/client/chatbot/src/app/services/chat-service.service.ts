import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: 'any',
})
export class ChatService {
  apiEndpoint:string = 'http://localhost:8000/ai';
  imageEndpoint:string = 'http://localhost:8000/image/';

  constructor(private http: HttpClient) { }

  generateErgonomicMessage(input:string) {
   return this.http.post(this.apiEndpoint, {"message":input, doImage : false});
  }

  getErgonomicImage(image_name: string) {
    return this.http.get(this.imageEndpoint + image_name);
  }
}
