package callhandler

import (
	"net/http"
	"os"
	"time"

	"hospital-service/internal/utils"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt/v5"
)

type StringeeHandler struct{}

func NewStringeeHandler() *StringeeHandler {
	return &StringeeHandler{}
}

// ---------------- Get Stringee Token ----------------
// @Summary Get Stringee access token
// @Tags Calls
// @Produce json
// @Param userId query string true "User ID"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /call/stringee-token [get]
func (h *StringeeHandler) GetStringeeToken(c *gin.Context) {
	userId := c.Query("userId")
	if userId == "" {
		c.JSON(http.StatusBadRequest, utils.ErrorResponse(http.StatusBadRequest, "Missing userId"))
		return
	}

	appKeySid := os.Getenv("API_SID_KEY")
	appKeySecret := os.Getenv("API_SERECT_KEY")

	if appKeySid == "" || appKeySecret == "" {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Missing Stringee credentials"))
		return
	}

	now := time.Now()
	exp := now.Add(1 * time.Hour)                 // 1 giờ
	jti := appKeySid + "-" + now.Format("150405") // tạo jti duy nhất theo thời gian

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"jti":    jti,
		"iss":    appKeySid,
		"exp":    exp.Unix(),
		"userId": userId, // 👈 Quan trọng
	})

	token.Header["cty"] = "stringee-api;v=1"
	token.Header["typ"] = "JWT"

	tokenString, err := token.SignedString([]byte(appKeySecret))
	if err != nil {
		c.JSON(http.StatusInternalServerError, utils.ErrorResponse(http.StatusInternalServerError, "Failed to sign token: "+err.Error()))
		return
	}

	c.JSON(http.StatusOK, utils.SuccessResponse(http.StatusOK, "Stringee token created", map[string]interface{}{
		"token": tokenString,
	}))
}
