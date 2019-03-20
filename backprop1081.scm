;; -*- coding: utf-8 -*-
;;
;; backprop1081.scm
;; 2019-3-20 v2.20
;;
;; ＜内容＞
;;   パラメータ設定ファイル
;;     学習する関数を mod(x, 2) - 1 にした (のこぎり波)
;;
(define outfile        "backprop_result1081.txt") ; 出力ファイル名

(define input-data-0   (lrange 0 2pi 0.1))      ; 入力(リスト)
(define correct-data-0 (map                     ; 正解(リスト)
                        (lambda (x1) (- (mod (/. (- x1 pi) pi) 2) 1))
                        input-data-0))
(define n-data         (length input-data-0))   ; データ数

(define input-data     (apply f-array           ; 入力(行列(1 x n-data))
                              0 1 0 n-data
                              ;; (入力の範囲を -1.0～1.0 に変換)
                              (map (lambda (x1) (/ (- x1 pi) pi)) input-data-0)))
(define correct-data   (apply f-array           ; 正解(行列(1 x n-data))
                              0 1 0 n-data
                              correct-data-0))

(define n-in           1)     ; 入力層のニューロン数
(define n-mid          50)    ; 中間層のニューロン数
(define n-out          1)     ; 出力層のニューロン数

(define ml-num         2)     ; 中間層の数
(define ml-func        'relu) ; 中間層の活性化関数(sigmoid / relu)

(define wb-width       0.01)  ; 重みとバイアスの幅
(define eta            0.1)   ; 学習係数
(define epoch          4001)  ; エポック数
(define interval       200)   ; 経過の表示間隔

