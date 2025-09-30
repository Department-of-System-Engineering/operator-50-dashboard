import {
  AfterViewChecked,
  AfterViewInit,
  Component,
  ElementRef, EventEmitter,
  OnInit, Output,
  ViewChild,
} from '@angular/core';

@Component({
  selector: 'app-chat-content',
  templateUrl: './chat-content.component.html',
  styleUrls: ['./chat-content.component.css'],
})
export class ChatContentComponent
{
  @Output() sendMessage: EventEmitter<string> = new EventEmitter();
  message: string = '';
  constructor(
  ) {}

  sendMessageToParentComponent(){
    this.sendMessage.emit(this.message);
  }
}
