import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app.routes';
import { AppComponent } from './app.component';
import { NavbarComponent } from './navbar/navbar.component';
import { HomeComponent } from './components/home-component/home.component';
import { AboutComponent } from './components/about-component/about.component';
import { ChatComponent } from './components/home-component/chat/chat.component';
import { ChatContentComponent } from './components/home-component/chat/chat-content/chat-content.component';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import {ChatListItemComponent} from './components/home-component/chat/chat-list-item/chat-list-item.component';
import {FormsModule} from '@angular/forms';
import {ChatService} from './services/chat-service.service';
import {HttpClient, HttpClientModule} from '@angular/common/http';
import {MarkdownComponent} from 'ngx-markdown';
import {SocketIoConfig, SocketIoModule} from 'ngx-socket-io';
const config: SocketIoConfig = { url: 'http://127.0.0.1:8000/', options: {transports: ["websocket"],autoConnect:false} };


@NgModule({
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    MarkdownComponent,
    SocketIoModule.forRoot(config)
  ],
  declarations: [
    NavbarComponent,
    HomeComponent,
    AboutComponent,
    AppComponent,
    ChatComponent,
    ChatContentComponent,
    ChatListItemComponent
  ],
  providers: [
    provideAnimationsAsync()
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
