package middlewares

import (
	"hospital-service/internal/services/subscriptionservice"
	"hospital-service/internal/utils"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// CheckSubscriptionLimit middleware kiểm tra giới hạn subscription trước khi sử dụng tính năng
func CheckSubscriptionLimit(subscriptionService *subscriptionservice.SubscriptionService, checkType string) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Lấy userID từ query hoặc body
		userID := c.Query("userId")
		if userID == "" {
			// Thử lấy từ body nếu là POST/PUT
			if c.Request.Method == "POST" || c.Request.Method == "PUT" {
				// Đối với form-data, lấy từ PostForm
				userID = c.PostForm("user_id")
				if userID == "" {
					// Đối với JSON, cần parse body (nhưng không thể đọc body 2 lần)
					// Vì vậy sẽ check sau khi handler xử lý
					c.Next()
					return
				}
			} else {
				c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "userId is required"))
				c.Abort()
				return
			}
		}

		var checkResult *subscriptionservice.CheckLimitResponse
		var err error

		if strings.ToLower(checkType) == "ai" {
			checkResult, err = subscriptionService.CheckAILimit(userID)
		} else if strings.ToLower(checkType) == "consult" {
			checkResult, err = subscriptionService.CheckConsultLimit(userID)
		} else {
			c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Invalid check type"))
			c.Abort()
			return
		}

		if err != nil {
			c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, err.Error()))
			c.Abort()
			return
		}

		if !checkResult.CanUse {
			c.JSON(http.StatusForbidden, utils.ErrorResponse(http.StatusForbidden, "Subscription limit exceeded"))
			c.Abort()
			return
		}

		// Lưu checkResult vào context để handler có thể sử dụng
		c.Set("subscription_check", checkResult)
		c.Set("user_id", userID)
		c.Next()
	}
}
