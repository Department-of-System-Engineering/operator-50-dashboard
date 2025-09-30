import { Component, OnInit } from '@angular/core';
import { ChatItem } from './chat.item';
import { ChatService } from '../../../services/chat-service.service';
import { Socket } from 'ngx-socket-io';
import { map } from 'rxjs';
import { SocketData } from './socketData';

@Component({
  selector: 'app-chat',
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent implements OnInit {
  messages: ChatItem[] = [];
  username: string = ''; // Felhasználó által megadott név
  isConnected: boolean = false; // Csatlakozási állapot

  constructor(private chatService: ChatService, private socket: Socket) {}

  ngOnInit() {}

  connectToSocket() {
    if (this.username.trim() === '') {
      alert('Kérlek, add meg a neved!');
      return;
    }

    this.socket.connect(); // Socket csatlakoztatása
    this.isConnected = true;

    // Név elküldése a szervernek
    this.socket.emit('setUsername', { username: this.username });
    this.getMessage().subscribe(); // Call the method and subscribe to the observable
  }

  getMessage() {
    return this.socket.fromEvent<SocketData>('message').pipe(
      map((data: SocketData) => {
        console.log(data);
        this.messages.push(<ChatItem>{ role: 'user', message: data.data });
      })
    );
  }

  sendMessage(message: string) {
    if (!this.isConnected) {
      alert('Először csatlakozz a sockethez!');
      return;
    }

    this.messages.push({ role: 'user', message: message });
    this.messages.push({ role: 'loading', message: '' });
    this.chatService.generateErgonomicMessage(message).subscribe((data) => {
      this.messages.pop();
      this.messages.push({ role: 'assistant', message: data.toString() });
      this.chatService.getErgonomicImage('positure.png').subscribe((image) => {
        console.log(image);
      });
    });
  }
}
