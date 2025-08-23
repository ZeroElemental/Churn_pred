#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#define BUFFER_SIZE 10
char a[BUFFER_SIZE];
int data = 97, front = 0, rear = 0, count = 0;
sem_t full, empty, mutex;
void *server(void *);
void *client(void *);
int main()
{

    pthread_t serverthread;
    pthread_t clientthread;
    sem_init(&full, 0, 0);
    sem_init(&empty, 0, BUFFER_SIZE);
    sem_init(&mutex, 0, 1);
    pthread_create(&serverthread, NULL, server, NULL);
    pthread_create(&clientthread, NULL, client, NULL);
    pthread_exit(NULL);
    return 0;
}
void *server(void *t)
{
    int r;
    while (1)
    {
        printf("\\nServer wants to produce item");
        sem_wait(&empty);
        sem_wait(&mutex);
        a[rear] = data;
        rear = (rear + 1) % BUFFER_SIZE;
        count++;
printf("\nServer produces %c & total element in
buffer %d",data,count);
 data++;
sem_post(&mutex);
 sem_post(&full);
 r=rand();
r=r%7+1;
sleep(r);
    }
    return (NULL);
}
void *client(void *e)
{
    while (1)
    {
        int r;
        printf("\nClient wants to consume");
        sem_wait(&full);
        sem_wait(&mutex);
        front = (front + 1) % BUFFER_SIZE;
        count--;
        printf("\nClient consume the data %c and convert it to %c. Now total buffer in buffer is %d", a[front - 1], toupper(a[front - 1]), count);
        sem_post(&mutex);
        sem_post(&empty);
        r = rand();
        r = r % 7 + 1;
        sleep(r);
    }

    return (NULL);
}
