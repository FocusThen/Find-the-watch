# Find the watch 

### Train and Query image

```sh
image1 = cv2.imread('../images/watcher.jpg')
image2 = cv2.imread('../images/watch.jpeg')

train_img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
query_img = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
```
![ss1](https://user-images.githubusercontent.com/47830409/64035279-0a7cf500-cb59-11e9-8cf4-6b9e02fde8f3.PNG)

### Find the Keys with ORB

  - More Details [ORB](https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html)
  
```sh
train_gray = cv2.cvtColor(train_img, cv2.COLOR_RGB2GRAY)
query_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(5000, 2.0)

keypoints_train, descriptors_train = orb.detectAndCompute(train_gray, None)
keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

query_img_keyp = copy.copy(query_img)

cv2.drawKeypoints(query_img, keypoints_query, query_img_keyp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(query_img_keyp)
plt.show()
```
![ss2](https://user-images.githubusercontent.com/47830409/64035383-4748ec00-cb59-11e9-96c2-ffc04d87b9bf.PNG)

### Matches

```sh
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(descriptors_train, descriptors_query)

matches = sorted(matches, key = lambda x : x.distance)

result = cv2.drawMatches(train_gray, keypoints_train, query_gray, keypoints_query, matches[:85], query_gray, flags = 2)

plt.title('Best Matching Points', fontsize = 30)
plt.imshow(result)
plt.show()
```

![ss3](https://user-images.githubusercontent.com/47830409/64035412-63e52400-cb59-11e9-9cbb-3167acda126d.PNG)
