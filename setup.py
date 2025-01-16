from setuptools import setup, find_packages

setup(
    name='trackeval',  # パッケージ名
    version='1.0.dev1',  # バージョン
    packages=find_packages(),  # `trackeval` フォルダを含む全てのサブモジュールを検出
    install_requires=[
        'numpy',
        'scipy',
    ],  # 必要な依存パッケージ
    description='TrackEval: A library for evaluating tracking algorithms.',
    author='Your Name',  # 必要に応じて
    author_email='your_email@example.com',  # 必要に応じて
    url='https://github.com/IIM-TTIJ/TrackEval',  # リポジトリのURL（オプション）
)