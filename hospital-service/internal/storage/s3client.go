package storage

import (
	"context"
	"mime/multipart"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

type S3Client struct {
	client *s3.Client
	bucket string
	region string
}

// NewS3Client khởi tạo S3Client từ config app
func NewS3Client(bucket, region, accessKey, secretKey string) (*S3Client, error) {
    cfg, err := config.LoadDefaultConfig(context.TODO(),
        config.WithRegion(region),
        config.WithCredentialsProvider(
            credentials.NewStaticCredentialsProvider(accessKey, secretKey, ""),
        ),
    )
    if err != nil {
        return nil, err
    }

    return &S3Client{
        client: s3.NewFromConfig(cfg),
        bucket: bucket,
        region: region,
    }, nil
}

// UploadFile upload trực tiếp stream lên S3
func (s *S3Client) UploadFile(fileHeader *multipart.FileHeader, key string) (string, error) {
	file, err := fileHeader.Open()
	if err != nil {
		return "", err
	}
	defer file.Close()

	_, err = s.client.PutObject(context.TODO(), &s3.PutObjectInput{
		Bucket:      aws.String(s.bucket),
		Key:         aws.String(key),
		Body:        file,
		ContentType: aws.String(fileHeader.Header.Get("Content-Type")),
	})
	if err != nil {
		return "", err
	}

	// Trả về public URL (giả sử bucket public hoặc có CloudFront)
	url := "https://" + s.bucket + ".s3." + s.region + ".amazonaws.com/" + key
	return url, nil
}
