# Scale pixel values from 0-255 
        img_array = img_array / 255

        # Reshape from 4D to 3D for PIL Image
        img_array3D = img_array.reshape(28,28,1)

        # Convert data type   
        img_array3D = img_array3D.astype('uint8')

        # Display image
        img = Image.fromarray(img_array3D)  
        img = img.resize((200,200))
        photo = ImageTk.PhotoImage(img)
        self.image['image'] = photo  
        self.image.image = photo