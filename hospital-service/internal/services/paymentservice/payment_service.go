package paymentservice

import (
	"crypto/hmac"
	"crypto/sha512"
	"encoding/hex"
	"hospital-service/internal/config"
	"net/url"
	"sort"
	"strconv"
	"time"
)

type VnpayService struct {
	tmnCode   string
	secretKey string
	vnpURL    string
	returnURL string
}

func NewVnpayService(cfg config.Config) *VnpayService {
	return &VnpayService{
		tmnCode:   cfg.VnpTmnCode,
		secretKey: cfg.VnpSecretKey,
		vnpURL:    cfg.VnpUrl,
		returnURL: cfg.VnpReturnUrl,
	}
}

func (s *VnpayService) CreatePaymentURL(amount int, orderId string) (string, error) {
	vnpParams := map[string]string{
		"vnp_Version":    "2.1.0",
		"vnp_Command":    "pay",
		"vnp_TmnCode":    s.tmnCode,
		"vnp_Locale":     "vn",
		"vnp_CurrCode":   "VND",
		"vnp_TxnRef":     orderId,
		"vnp_OrderInfo":  "Thanh toan don hang",
		"vnp_OrderType":  "other",
		"vnp_Amount":     strconv.Itoa(amount * 100),
		"vnp_ReturnUrl":  s.returnURL + "?orderId=" + orderId,
		"vnp_IpAddr":     "127.0.0.1",
		"vnp_CreateDate": time.Now().Format("20060102150405"),
	}

	// --- sort keys ---
	keys := make([]string, 0, len(vnpParams))
	for k := range vnpParams {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// --- build query string ENCODED ---
	values := url.Values{}
	for _, k := range keys {
		values.Add(k, vnpParams[k])
	}
	queryString := values.Encode() // đã url-encode giống Node

	// --- signData = queryString ---
	h := hmac.New(sha512.New, []byte(s.secretKey))
	h.Write([]byte(queryString))
	secureHash := hex.EncodeToString(h.Sum(nil))

	// --- return final URL ---
	return s.vnpURL + "?" + queryString + "&vnp_SecureHash=" + secureHash, nil
}

// CreatePaymentURLForSubscription tạo payment URL cho subscription
func (s *VnpayService) CreatePaymentURLForSubscription(amount int, subscriptionID, userID, planName string, duration int) (string, error) {
	// Tạo return URL với các thông tin subscription
	returnURL := s.returnURL + "?type=subscription&subscriptionId=" + url.QueryEscape(subscriptionID) +
		"&userId=" + url.QueryEscape(userID) +
		"&planName=" + url.QueryEscape(planName) +
		"&duration=" + strconv.Itoa(duration)

	vnpParams := map[string]string{
		"vnp_Version":    "2.1.0",
		"vnp_Command":     "pay",
		"vnp_TmnCode":     s.tmnCode,
		"vnp_Locale":      "vn",
		"vnp_CurrCode":    "VND",
		"vnp_TxnRef":      subscriptionID,
		"vnp_OrderInfo":   "Thanh toan goi " + planName,
		"vnp_OrderType":   "other",
		"vnp_Amount":      strconv.Itoa(amount * 100),
		"vnp_ReturnUrl":   returnURL,
		"vnp_IpAddr":      "127.0.0.1",
		"vnp_CreateDate":  time.Now().Format("20060102150405"),
	}

	// --- sort keys ---
	keys := make([]string, 0, len(vnpParams))
	for k := range vnpParams {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// --- build query string ENCODED ---
	values := url.Values{}
	for _, k := range keys {
		values.Add(k, vnpParams[k])
	}
	queryString := values.Encode()

	// --- signData = queryString ---
	h := hmac.New(sha512.New, []byte(s.secretKey))
	h.Write([]byte(queryString))
	secureHash := hex.EncodeToString(h.Sum(nil))

	// --- return final URL ---
	return s.vnpURL + "?" + queryString + "&vnp_SecureHash=" + secureHash, nil
}

func (s *VnpayService) VerifyReturn(query url.Values) bool {
	vnpSecureHash := query.Get("vnp_SecureHash")
	// Tạo copy để không ảnh hưởng đến query gốc
	queryCopy := make(url.Values)
	for k, v := range query {
		queryCopy[k] = v
	}
	
	queryCopy.Del("vnp_SecureHash")
	// Xóa các params không phải của VNPay
	queryCopy.Del("orderId")
	queryCopy.Del("type")
	queryCopy.Del("subscriptionId")
	queryCopy.Del("userId")
	queryCopy.Del("planName")
	queryCopy.Del("duration")

	// sort keys
	keys := make([]string, 0, len(queryCopy))
	for k := range queryCopy {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	values := url.Values{}
	for _, k := range keys {
		values.Add(k, queryCopy.Get(k))
	}
	queryString := values.Encode()

	h := hmac.New(sha512.New, []byte(s.secretKey))
	h.Write([]byte(queryString))
	hash := hex.EncodeToString(h.Sum(nil))

	return hash == vnpSecureHash
}
