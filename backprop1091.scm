;; -*- coding: utf-8 -*-
;;
;; backprop1091.scm
;; 2019-3-13 v2.00
;;
;; ＜内容＞
;;   学習する関数を (2 / pi) * asin(sin(x)) にした (三角波)
;;
(define outfile        "backprop_result1091.txt") ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map                     ; 正解(リスト)
                        (lambda (x1) (* 2 1/pi (%asin (%sin x1))))
                        input-data-0))
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f64array-simple   ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (map (lambda (x1) (/ (- x1 pi) pi)) input-data-0)))
(define correct-data   (apply f64array-simple   ; 正解(行列(1 x n-data))
                              0 1 0 n-data
                              correct-data-0))

(define n-in           1)     ; 入力層のニューロン数
(define n-mid          20)    ; 中間層のニューロン数
(define n-out          1)     ; 出力層のニューロン数

(define ml-num         1)     ; 中間層の数
(define ml-func        'relu) ; 中間層の活性化関数(sigmoid / relu)

(define wb-width       0.01)  ; 重みとバイアスの幅
(define eta            0.1)   ; 学習係数
(define epoch          2001)  ; エポック数
(define interval       200)   ; 経過の表示間隔

