import {Component, Input} from '@angular/core';
import {marked} from 'marked';


@Component({
  selector: 'app-chat-list-item',
  templateUrl: './chat-list-item.component.html',
  styleUrl: './chat-list-item.component.css'
})
export class ChatListItemComponent {
  @Input() aiRole: string = "";
  @Input() message: string = "";

  get renderedMessage(): string {
    const result = marked(this.message || '');
    if (result instanceof Promise) {
      console.error('Async Markdown rendering is not supported in this context.');
      return '';
    }
    return result;
  }

}
