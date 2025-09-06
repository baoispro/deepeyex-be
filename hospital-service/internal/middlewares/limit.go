package middlewares

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// LimitRequestBody middleware giới hạn dung lượng request body
func LimitRequestBody(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}
